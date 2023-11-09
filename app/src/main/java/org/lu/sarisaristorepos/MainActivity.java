package org.lu.sarisaristorepos;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.widget.Button;

public class MainActivity extends AppCompatActivity {

    Button pos, product, logout;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        pos = findViewById(R.id.posBtn);
        product = findViewById(R.id.productsBtn);
        logout = findViewById(R.id.logoutBtn);

        pos.setOnClickListener(v -> PointOfSale());
        product.setOnClickListener(v -> Product());
        logout.setOnClickListener(v -> finish());

    }

    private void Product() {
        Intent intent = new Intent(this, ProductsActivity.class);
        startActivity(intent);
    }

    private void PointOfSale() {
        Intent intent = new Intent(this, PointOfSaleActivity.class);
        startActivity(intent);

    }
}